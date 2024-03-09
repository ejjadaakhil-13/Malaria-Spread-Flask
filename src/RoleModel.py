from Constants import connString
import pyodbc
import datetime
import uuid
import time    

class RoleModel:
    def __init__(self, roleID = 0,roleName = '',canRole = False,canUsers = False,canOwner = False,canProperty = False,canPropertyType = False,canRentalAgreement = False,canTenant = False):
        self.roleID = roleID
        self.roleName = roleName
        self.canRole = canRole
        self.canUsers = canUsers
        self.canOwner = canOwner
        self.canProperty = canProperty
        self.canPropertyType = canPropertyType
        self.canRentalAgreement = canRentalAgreement
        self.canTenant = canTenant
       
        

    @staticmethod
    def get_all():
        conn = pyodbc.connect(connString, autocommit=True)
        cursor = conn.cursor()
        sqlcmd1 = "SELECT * FROM Role ORDER BY roleName"
        cursor.execute(sqlcmd1)
        records = []
        for dbrow in cursor.fetchall():
            row = RoleModel(dbrow[0],dbrow[1],dbrow[2],dbrow[3],dbrow[4],dbrow[5],dbrow[6],dbrow[7],dbrow[8])
            records.append(row)
        cursor.close()
        conn.close()
        return records

    @staticmethod
    def get_name_id():
        conn = pyodbc.connect(connString, autocommit=True)
        cursor = conn.cursor()
        sqlcmd1 = "SELECT roleID, roleName FROM Role ORDER BY roleName"
        cursor.execute(sqlcmd1)
        records = []
        for dbrow in cursor.fetchall():
            row = RoleModel(dbrow[0],dbrow[1])
            records.append(row)
        cursor.close()
        conn.close()
        return records
        
    @staticmethod
    def get_by_id(unique_id):
        conn = pyodbc.connect(connString, autocommit=True)
        cursor = conn.cursor()
        sqlcmd1 = "SELECT * FROM Role WHERE roleID = ?"
        cursor.execute(sqlcmd1, unique_id)
        record = None
        for dbrow in cursor.fetchall():
            record = RoleModel(dbrow[0],dbrow[1],dbrow[2],dbrow[3],dbrow[4],dbrow[5],dbrow[6],dbrow[7],dbrow[8])
        cursor.close()
        conn.close()
        return record
    
    @staticmethod
    def insert(obj):
        obj.roleID = str(uuid.uuid4())
        conn = pyodbc.connect(connString, autocommit=True)
        cursor = conn.cursor()
        sqlcmd1 = "INSERT INTO Role (roleName,canRole,canUsers,canOwner,canProperty,canPropertyType,canRentalAgreement,canTenant) VALUES(?,?,?,?,?,?,?,?)"
        cursor.execute(sqlcmd1, (obj.roleName,obj.canRole,obj.canUsers,obj.canOwner,obj.canProperty,obj.canPropertyType,obj.canRentalAgreement,obj.canTenant))
        cursor.close()
        conn.close()
        
    
    @staticmethod
    def update(obj):
        conn = pyodbc.connect(connString, autocommit=True)
        cursor = conn.cursor()
        sqlcmd1 = "UPDATE Role SET roleName = ?,canRole = ?,canUsers = ?,canOwner = ?,canProperty = ?,canPropertyType = ?,canRentalAgreement = ?,canTenant = ? WHERE roleID = ?"
        cursor.execute(sqlcmd1,  (obj.roleName,obj.canRole,obj.canUsers,obj.canOwner,obj.canProperty,obj.canPropertyType,obj.canRentalAgreement,obj.canTenant,obj.roleID))
        cursor.close()
        conn.close()
    
    @staticmethod
    def delete(unique_id):
        conn = pyodbc.connect(connString, autocommit=True)
        cursor = conn.cursor()
        sqlcmd1 = "DELETE FROM Role WHERE roleID = ?"
        cursor.execute(sqlcmd1, (unique_id))
        cursor.close()
        conn.close()

